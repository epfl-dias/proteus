import { ComponentFixture, TestBed } from '@angular/core/testing';

import { BwProfileComponent } from './bw-profile.component';

describe('BwProfileComponent', () => {
  let component: BwProfileComponent;
  let fixture: ComponentFixture<BwProfileComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ BwProfileComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(BwProfileComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
